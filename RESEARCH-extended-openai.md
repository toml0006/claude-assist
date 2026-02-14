# Extended OpenAI Conversation - Research Summary

## Overview

**Repository**: https://github.com/jekalmin/extended_openai_conversation  
**Purpose**: A Home Assistant custom component that extends the standard OpenAI Conversation integration with powerful custom tool/function capabilities.

## Architecture

### Key Differences from Standard HA Conversation

| Standard HA Approach | Extended OpenAI Approach |
|---------------------|-------------------------|
| Uses `llm.Tool` class registration | Uses YAML-defined function specs |
| Tools auto-discovered via LLM API | Custom functions defined in config options |
| Exposes HA services via `AssistAPI` | Implements own function executor system |
| Built into conversation component | Standalone custom component |

### Component Structure

```
custom_components/extended_openai_conversation/
├── __init__.py          # Setup & config entry
├── config_flow.py       # Integration setup UI
├── const.py             # Constants & default function specs
├── conversation.py      # ConversationEntity implementation
├── entity.py            # Base LLM entity with streaming
├── helpers.py           # FunctionExecutor classes (CORE!)
├── services.py          # HA service definitions
└── template.py          # Template rendering utilities
```

### Core Design Pattern

The integration does NOT use HA's `llm.Tool` class. Instead, it:

1. **Defines functions as YAML specs** stored in config entry options
2. **Converts specs to OpenAI tool format** at runtime
3. **Implements custom FunctionExecutor classes** to handle each function type
4. **Manages the tool call loop** itself (not using HA's built-in handling)

---

## Function Types & Executors

### Registry Pattern

```python
FUNCTION_EXECUTORS: dict[str, FunctionExecutor] = {
    "native": NativeFunctionExecutor(),
    "script": ScriptFunctionExecutor(),
    "template": TemplateFunctionExecutor(),
    "rest": RestFunctionExecutor(),
    "scrape": ScrapeFunctionExecutor(),
    "composite": CompositeFunctionExecutor(),
    "sqlite": SqliteFunctionExecutor(),
}
```

### 1. Native Functions

Built-in functions that access HA internals directly:

| Function Name | Description |
|--------------|-------------|
| `execute_service` | Call HA services with entity/area/device targets |
| `execute_service_single` | Single service call (internal) |
| `add_automation` | Create automations from YAML config |
| `get_history` | Retrieve state history from recorder |
| `get_energy` | Get energy manager configuration |
| `get_statistics` | Get recorder statistics (energy, sensors) |
| `get_user_from_user_id` | Map user_id to friendly name |

#### `execute_service` Implementation

```python
async def execute_service_single(
    self, hass, function, service_argument, llm_context, exposed_entities
):
    domain = service_argument["domain"]
    service = service_argument["service"]
    service_data = service_argument.get("service_data", service_argument.get("data", {}))
    entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
    
    # Validate entity exists and is exposed
    self.validate_entity_ids(hass, entity_id or [], exposed_entities)
    
    # Verify service exists
    if not hass.services.has_service(domain, service):
        raise ServiceNotFound(domain, service)
    
    await hass.services.async_call(
        domain=domain,
        service=service,
        service_data=service_data,
    )
    return {"success": True}
```

#### `get_history` Implementation

```python
async def get_history(self, hass, function, arguments, llm_context, exposed_entities):
    start_time = arguments.get("start_time")
    end_time = arguments.get("end_time")
    entity_ids = arguments.get("entity_ids", [])
    
    # Validate entities are exposed
    self.validate_entity_ids(hass, entity_ids, exposed_entities)
    
    # Query recorder
    with recorder.util.session_scope(hass=hass, read_only=True) as session:
        result = await recorder.get_instance(hass).async_add_executor_job(
            recorder_history.get_significant_states_with_session,
            hass, session, start_time, end_time, entity_ids,
            None,  # filters
            include_start_time_state,
            significant_changes_only,
            minimal_response,
            no_attributes,
        )
    return [[self.as_dict(item) for item in sublist] for sublist in result.values()]
```

#### `get_statistics` Implementation

```python
async def get_statistics(self, hass, function, arguments, llm_context, exposed_entities):
    statistic_ids = arguments.get("statistic_ids", [])
    start_time = dt_util.as_utc(dt_util.parse_datetime(arguments["start_time"]))
    end_time = dt_util.as_utc(dt_util.parse_datetime(arguments["end_time"]))
    
    return await recorder.get_instance(hass).async_add_executor_job(
        recorder.statistics.statistics_during_period,
        hass,
        start_time,
        end_time,
        statistic_ids,
        arguments.get("period", "day"),
        arguments.get("units"),
        arguments.get("types", {"change"}),
    )
```

#### `add_automation` Implementation

```python
async def add_automation(self, hass, function, arguments, llm_context, exposed_entities):
    automation_config = yaml.safe_load(arguments["automation_config"])
    config = {"id": str(round(time.time() * 1000))}
    config.update(automation_config[0] if isinstance(automation_config, list) else automation_config)
    
    # Validate automation config
    await _async_validate_config_item(hass, config, True, False)
    
    # Append to automations.yaml
    with open(os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH), "a") as f:
        raw_config = yaml.dump([config], allow_unicode=True, sort_keys=False)
        f.write("\n" + raw_config)
    
    # Reload automations
    await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)
    
    # Fire event for tracking
    hass.bus.async_fire(EVENT_AUTOMATION_REGISTERED, {"automation_config": config})
    return "Success"
```

### 2. Script Function Executor

Executes HA script sequences:

```python
class ScriptFunctionExecutor(FunctionExecutor):
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        script = Script(
            hass,
            function["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description="[extended_openai_conversation] function",
        )
        
        context = llm_context.context if llm_context else None
        result = await script.async_run(run_variables=arguments, context=context)
        
        # Return custom result if set
        if result is None:
            return "Success"
        return result.variables.get("_function_result", "Success")
```

**Usage Example**:
```yaml
- spec:
    name: add_item_to_shopping_cart
    parameters:
      type: object
      properties:
        item:
          type: string
  function:
    type: script
    sequence:
    - service: shopping_list.add_item
      data:
        name: '{{item}}'
```

### 3. Template Function Executor

Renders Jinja2 templates with arguments:

```python
class TemplateFunctionExecutor(FunctionExecutor):
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        return function["value_template"].async_render(
            arguments,
            parse_result=function.get("parse_result", False),
        )
```

**Usage Example**:
```yaml
- spec:
    name: get_attributes
  function:
    type: template
    value_template: |
      {% for entity in entity_id %}
        {{entity}}: {{states[entity].attributes}}
      {% endfor %}
```

### 4. REST Function Executor

Makes HTTP requests:

```python
class RestFunctionExecutor(FunctionExecutor):
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        rest_data = _get_rest_data(hass, function, arguments)
        await rest_data.async_update()
        value = rest_data.data_without_xml()
        
        # Apply value_template if provided
        if value_template := function.get("value_template"):
            value = value_template.async_render_with_possible_json_value(value, None, arguments)
        return value
```

### 5. Scrape Function Executor

Scrapes web pages with CSS selectors:

```python
class ScrapeFunctionExecutor(FunctionExecutor):
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        rest_data = _get_rest_data(hass, function, arguments)
        coordinator = scrape.coordinator.ScrapeCoordinator(hass, None, rest_data, function)
        await coordinator.async_refresh()
        
        # Extract values using CSS selectors
        for sensor_config in function["sensor"]:
            value = self._extract_value(coordinator.data, sensor_config)
            # ... process with templates
```

### 6. Composite Function Executor

Chains multiple functions together:

```python
class CompositeFunctionExecutor(FunctionExecutor):
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        sequence = function["sequence"]
        new_arguments = arguments.copy()
        
        for executor_config in sequence:
            function_executor = get_function_executor(executor_config["type"])
            result = await function_executor.execute(
                hass, executor_config, new_arguments, llm_context, exposed_entities
            )
            
            # Store result for next step
            if response_variable := executor_config.get("response_variable"):
                new_arguments[response_variable] = result
        
        return result
```

**Usage Example** (get_history with formatting):
```yaml
- spec:
    name: get_history
  function:
    type: composite
    sequence:
      - type: native
        name: get_history
        response_variable: history_result
      - type: template
        value_template: >-
          {% for item_list in history_result %}
            {% for item in item_list %}
              {{ item.last_changed }}: {{ item.state }}
            {% endfor %}
          {% endfor %}
```

### 7. SQLite Function Executor

Direct database queries (read-only):

```python
class SqliteFunctionExecutor(FunctionExecutor):
    def get_default_db_url(self, hass):
        db_file_path = os.path.join(hass.config.config_dir, recorder.DEFAULT_DB_FILE)
        return f"file:{db_file_path}?mode=ro"
    
    async def execute(self, hass, function, arguments, llm_context, exposed_entities):
        db_url = self.set_url_read_only(function.get("db_url", self.get_default_db_url(hass)))
        query = function.get("query", "{{query}}")
        
        # Render query with safety helpers
        template_arguments = {
            "is_exposed": lambda e: self.is_exposed(e, exposed_entities),
            "is_exposed_entity_in_query": lambda q: self.is_exposed_entity_in_query(q, exposed_entities),
            "exposed_entities": exposed_entities,
            "raise": self.raise_error,
            **arguments
        }
        
        q = Template(query, hass).async_render(template_arguments)
        
        with sqlite3.connect(db_url, uri=True) as conn:
            cursor = conn.cursor().execute(q)
            # Return results as list of dicts
```

---

## Tool Schema Exposure

### Default Function Spec (from `const.py`)

```python
DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Execute service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay": {
                        "type": "object",
                        "description": "Time to wait before execution",
                        "properties": {
                            "hours": {"type": "integer", "minimum": 0},
                            "minutes": {"type": "integer", "minimum": 0},
                            "seconds": {"type": "integer", "minimum": 0},
                        },
                    },
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {"type": "string"},
                                "service": {"type": "string"},
                                "service_data": {
                                    "type": "object",
                                    "properties": {
                                        "entity_id": {"type": "array", "items": {"type": "string"}},
                                        "area_id": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    },
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    },
    {
        "spec": {
            "name": "get_attributes",
            "description": "Get attributes of entity or multiple entities.",
            "parameters": {...}
        },
        "function": {
            "type": "template",
            "value_template": "..."
        },
    },
]
```

### Conversion to OpenAI Format

```python
# In entity.py
tools: list[ChatCompletionToolParam] = [
    ChatCompletionToolParam(
        type="function",
        function=func_spec["spec"],  # OpenAI JSON schema format
    )
    for func_spec in custom_functions
]
```

---

## Exposed Entity Handling

```python
def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get entities exposed to conversation."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    entity_registry = er.async_get(hass)
    
    return [
        {
            "entity_id": state.entity_id,
            "name": state.name,
            "state": state.state,
            "aliases": list(entity.aliases) if entity and entity.aliases else [],
        }
        for state in states
        for entity in [entity_registry.async_get(state.entity_id)]
    ]
```

---

## Adapting for ai-subscription-assist (llm.Tool Pattern)

### Key Insight

Extended OpenAI Conversation **bypasses** HA's `llm.Tool` system entirely. To achieve similar functionality in ai-subscription-assist while using the standard HA pattern, we need to:

1. **Register tools via `llm.Tool`** (as we currently do)
2. **Implement the same underlying functionality** in our tool implementations

### Migration Strategy

| Extended OpenAI Function | ai-subscription-assist Implementation |
|-------------------------|------------------------------|
| `native: execute_service` | Already handled by HA's `AssistAPI` |
| `native: get_history` | New `llm.Tool` → call recorder API |
| `native: get_statistics` | New `llm.Tool` → call recorder statistics |
| `native: add_automation` | New `llm.Tool` → write automations.yaml |
| `native: get_energy` | New `llm.Tool` → energy manager data |
| `native: get_user_from_user_id` | New `llm.Tool` → auth.async_get_user |
| `template` | Implement as `llm.Tool` with template rendering |
| `script` | Implement as `llm.Tool` wrapping Script class |
| `sqlite` | New `llm.Tool` → recorder database query |

### Example: get_history as llm.Tool

```python
from homeassistant.components.recorder import history as recorder_history
from homeassistant.components import recorder
from homeassistant.helpers import llm
import voluptuous as vol

class GetHistoryTool(llm.Tool):
    """Tool to retrieve entity state history."""
    
    name = "get_history"
    description = "Retrieve historical state data for entities."
    
    parameters = vol.Schema({
        vol.Required("entity_ids"): vol.All(cv.ensure_list, [cv.entity_id]),
        vol.Optional("start_time"): cv.datetime,
        vol.Optional("end_time"): cv.datetime,
        vol.Optional("significant_changes_only", default=True): bool,
        vol.Optional("minimal_response", default=True): bool,
    })
    
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> str:
        entity_ids = tool_input.tool_args["entity_ids"]
        start_time = tool_input.tool_args.get("start_time", dt_util.utcnow() - timedelta(days=1))
        end_time = tool_input.tool_args.get("end_time", start_time + timedelta(days=1))
        
        # Validate entities are exposed
        for entity_id in entity_ids:
            if not async_should_expose(hass, conversation.DOMAIN, entity_id):
                raise HomeAssistantError(f"Entity {entity_id} is not exposed")
        
        # Query recorder
        with recorder.util.session_scope(hass=hass, read_only=True) as session:
            result = await recorder.get_instance(hass).async_add_executor_job(
                recorder_history.get_significant_states_with_session,
                hass, session, start_time, end_time, entity_ids,
                None,  # filters
                True,   # include_start_time_state
                tool_input.tool_args.get("significant_changes_only", True),
                tool_input.tool_args.get("minimal_response", True),
                True,   # no_attributes
            )
        
        # Format for LLM consumption
        formatted = []
        for entity_id, states in result.items():
            formatted.append({
                "entity_id": entity_id,
                "states": [
                    {"state": s.state, "last_changed": s.last_changed.isoformat()}
                    for s in states
                ]
            })
        
        return json.dumps(formatted)
```

### Example: get_statistics as llm.Tool

```python
class GetStatisticsTool(llm.Tool):
    """Tool to retrieve recorder statistics."""
    
    name = "get_statistics"
    description = "Get statistics for sensors (energy, temperature, etc.) over time."
    
    parameters = vol.Schema({
        vol.Required("statistic_ids"): vol.All(cv.ensure_list, [str]),
        vol.Required("start_time"): cv.datetime,
        vol.Required("end_time"): cv.datetime,
        vol.Optional("period", default="day"): vol.In(["5minute", "hour", "day", "week", "month"]),
        vol.Optional("types", default=["change"]): [vol.In(["change", "last_reset", "max", "mean", "min", "state", "sum"])],
    })
    
    async def async_call(self, hass, tool_input, llm_context) -> str:
        args = tool_input.tool_args
        
        result = await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            dt_util.as_utc(args["start_time"]),
            dt_util.as_utc(args["end_time"]),
            args["statistic_ids"],
            args["period"],
            None,  # units
            set(args.get("types", ["change"])),
        )
        
        return json.dumps(result, default=str)
```

### Tool Registration

```python
# In api.py or wherever LLM API is set up
@callback
def async_register_api(hass: HomeAssistant):
    """Register the AI Subscription Assist LLM API."""
    
    async def async_get_tools(llm_context: llm.LLMContext) -> list[llm.Tool]:
        tools = []
        
        # Include standard HA Assist tools
        assist_api = llm.async_get_api(hass, "assist")
        if assist_api:
            tools.extend(await assist_api.async_get_tools(llm_context))
        
        # Add our custom tools
        tools.extend([
            GetHistoryTool(),
            GetStatisticsTool(),
            RenderTemplateTool(),
            ExecuteScriptTool(),
            QueryDatabaseTool(),
        ])
        
        return tools
    
    llm.async_register_api(hass, ClaudeAssistAPI(hass, async_get_tools))
```

---

## Summary: What to Implement

### Priority 1: Core Tools (Most Useful)

1. **`get_history`** - Entity state history
2. **`get_statistics`** - Recorder statistics (energy, sensors)  
3. **`render_template`** - Evaluate Jinja2 templates

### Priority 2: Advanced Tools

4. **`execute_script`** - Run script sequences
5. **`query_database`** - Read-only SQL queries on recorder DB
6. **`add_automation`** - Create automations via YAML

### Priority 3: Utility Tools

7. **`get_user_name`** - Map user_id to name
8. **`get_energy_config`** - Energy dashboard configuration
9. **`scrape_url`** - Web scraping with CSS selectors
10. **`fetch_url`** - REST API calls

### Key Implementation Notes

1. **Validation**: Always validate entities are exposed before returning data
2. **Read-only DB**: Force read-only mode on database connections
3. **Template safety**: Use `parse_result=False` for untrusted templates
4. **Error handling**: Wrap errors in user-friendly messages
5. **Context**: Pass `llm_context` through for user identification

---

## Files to Reference

- `/tmp/extended_openai_conversation/custom_components/extended_openai_conversation/helpers.py` - All FunctionExecutor implementations
- `/tmp/extended_openai_conversation/custom_components/extended_openai_conversation/entity.py` - Chat handling & tool execution
- `/tmp/extended_openai_conversation/custom_components/extended_openai_conversation/const.py` - Default function specs
- `/tmp/extended_openai_conversation/examples/` - Practical YAML examples

"""Conversation support for AI Subscription Assist."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import AiSubscriptionAssistConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import AiSubscriptionAssistBaseLLMEntity
from .memory_service import get_memory_service


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: AiSubscriptionAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [AiSubscriptionAssistConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class AiSubscriptionAssistConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    AiSubscriptionAssistBaseLLMEntity,
):
    """AI Subscription Assist conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: AiSubscriptionAssistConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.subentry.data
        memory_service = get_memory_service(self.hass, self.entry.entry_id)
        extra_system_prompt = user_input.extra_system_prompt

        if memory_service is not None:
            handled, command_response = await memory_service.async_handle_command(
                user_input,
                self.subentry.subentry_id,
            )
            if handled:
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content=command_response,
                    )
                )
                return conversation.async_get_result_from_chat_log(user_input, chat_log)

            await memory_service.async_inject_resume_context(
                chat_log,
                user_input,
                self.subentry.subentry_id,
                self.entity_id,
            )
            await memory_service.async_maybe_capture_heuristic(user_input)
            memory_prompt = await memory_service.async_build_memory_prompt(user_input)
            if memory_prompt:
                extra_system_prompt = "\n\n".join(
                    part for part in (extra_system_prompt, memory_prompt) if part
                )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)
        if memory_service is not None:
            assistant_text = None
            if chat_log.content and isinstance(chat_log.content[-1], conversation.AssistantContent):
                assistant_text = chat_log.content[-1].content
            await memory_service.async_record_turn(
                user_input,
                self.subentry.subentry_id,
                assistant_text,
            )
        return result

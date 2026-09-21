"""ChatGPT Web UI selectors and browser-integration constants.

This module defines the DOM contract used by the Playwright backend.
These values are implementation details of the ChatGPT Web adapter,
not operator configuration.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Core page elements
# ---------------------------------------------------------------------------

RATE_LIMIT_MODAL = (
    '[data-testid="modal-conversation-history-rate-limit"]'
)

COMPOSER = (
    "#prompt-textarea, div[contenteditable='true']"
)

ASSISTANT_MESSAGES = (
    "[data-message-author-role='assistant']"
)

BODY = "body"


# ---------------------------------------------------------------------------
# Temporary Chat / personalization
# ---------------------------------------------------------------------------

TEMPORARY_CHAT_TOGGLE = (
    'button[aria-label="Temporary chat"]'
)

TEMPORARY_CHAT_ENABLED = (
    'button[aria-label="Turn off temporary chat"]'
)

PERSONALIZATION_OPTION = (
    '[role="menuitemradio"]'
)


def aria_label_button(label: str) -> str:
    """Build the exact button selector used for trusted UI labels."""
    return f'button[aria-label="{label}"]'


# ---------------------------------------------------------------------------
# Model / reasoning picker
# ---------------------------------------------------------------------------

REASONING_CONTROLS = (
    "[aria-haspopup='menu']"
)

INTELLIGENCE_PICKER_CONTENT = (
    "[data-testid='composer-intelligence-picker-content']"
)

MODEL_PICKER_SIMPLE_VIEW = (
    "[data-testid='composer-model-picker-slider-simple-view']"
)

MODEL_PICKER_ADVANCED_VIEW = (
    "[data-testid='composer-model-picker-slider-advanced-view']"
)

SELECT_MODEL_MENUITEM = (
    "[role='menuitem'][aria-label='Select model']"
)

MODEL_OPTIONS = (
    "[role='menuitemradio']"
)

REASONING_SLIDER_SELECTORS = (
    "[role='slider']",
    "[aria-valuenow]",
    "[aria-valuetext]",
    "[tabindex='0']:not([aria-label='Select model'])",
)

PARENT_XPATH = "xpath=.."


# ---------------------------------------------------------------------------
# Submit / generation
# ---------------------------------------------------------------------------

SEND_BUTTON_SELECTORS = (
    "button[data-testid='send-button']",
    "button[aria-label='Send prompt']",
    "button[aria-label='Send message']",
    "button[aria-label='Send']",
)

STOP_BUTTON = (
    "button[aria-label*='Stop'], "
    "button[data-testid*='stop']"
)


# ---------------------------------------------------------------------------
# Conversation creation
# ---------------------------------------------------------------------------

NEW_CHAT_SELECTORS = (
    "[data-testid='create-new-chat-button']",
    "button[aria-label*='New chat']",
    "a[aria-label*='New chat']",
    "a[href='/']",
)


# ---------------------------------------------------------------------------
# Internal search bounds
#
# These are implementation constraints, not operator policy. Keep them here
# rather than exposing them as deployment configuration.
# ---------------------------------------------------------------------------

CONTROL_ANCESTOR_MAX_DEPTH = 8
UI_CANDIDATE_LIMIT = 20
SEND_BUTTON_CANDIDATE_LIMIT = 4


# ---------------------------------------------------------------------------
# Short DOM probe timing
#
# These are intentionally not provider configuration. They are small
# Playwright probe/poll intervals internal to the UI adapter.
# ---------------------------------------------------------------------------

CONTROL_VISIBLE_PROBE_MS = 100
CONTROL_TEXT_PROBE_MS = 200

OPTION_VISIBLE_PROBE_MS = 150
OPTION_TEXT_PROBE_MS = 250

PICKER_OPEN_PROBE_MS = 500
PICKER_TEXT_TIMEOUT_MS = 1_000
PICKER_STATE_TEXT_TIMEOUT_MS = 500

STATE_POLL_MS = 100
TRANSITION_SETTLE_MS = 200

SEND_BUTTON_PROBE_MS = 250
SUBMIT_STOP_PROBE_MS = 200
SUBMIT_COMPOSER_PROBE_MS = 500
SUBMIT_POLL_MS = 200

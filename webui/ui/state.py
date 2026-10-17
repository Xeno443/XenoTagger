"""Process-wide app config shared by every tab. current_cfg is REBOUND
(not mutated in place) by Save settings and the Models tab's "Set as
active model", so always read it as `state.current_cfg` at call time - a
`from ui.state import current_cfg` would freeze whatever it was at import.
"""

from __future__ import annotations

import threading

from core import config as config_mod
from core.config import AppConfig

current_cfg: AppConfig = config_mod.load()

# Guards the read-modify-write of current_cfg + config_mod.save() in
# save_settings_ui() and models_set_active_ui() - the only two places that
# mutate current_cfg wholesale. Without it, two overlapping saves (e.g.
# two browser tabs) could each read a stale current_cfg and the second
# write would silently clobber fields the first one just changed. Plain
# Lock, not RLock like the session/operation locks - neither function calls back into
# itself or the other while already holding it, so there's no reentrancy
# need.
config_lock = threading.Lock()

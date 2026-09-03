# Debuglog tab

Raw log output for troubleshooting. Hidden by default - enable it from
[Settings – Debug](tab-settings-debug.md) and restart the app to make
this tab appear.

![Debuglog tab overview](images/debuglog-overview.png)

## Python debug log

![Python debug log box](images/debuglog-python.png)

The application's own internal log, newest entries at the top. Also
written continuously to a log file on disk regardless of whether this
tab is open.

## llama-server output

![llama-server output box](images/debuglog-llama.png)

Raw stdout/stderr from the managed `llama-server` process. Reads `n/a`
when running in External mode, since there is no managed process to
capture output from - check that server's own logs instead.

## Clear

**Clear** empties both boxes shown above (and the on-disk log files, for
the managed server's log).

## Related

- [Settings – Debug](tab-settings-debug.md) - the switch that makes this tab visible.
- [Settings – Llama](tab-settings-llama.md) - server mode, which determines whether the llama-server box has content.

---

[← Settings – Debug](tab-settings-debug.md) · [Manual home](README.md)

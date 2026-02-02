# S30 - cloud-websocket | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S30
  name: cloud-websocket
  title: "WebSocket Real-time"
  wave: W3-CloudMVP
  priority: P2-MEDIUM
  type: implementation

objective: "WebSocket para validações real-time"

dependencies:
  - S28  # Cloud API Base

deliverables:
  - src/hl_mcp/cloud/ws/__init__.py
  - src/hl_mcp/cloud/ws/manager.py
  - src/hl_mcp/cloud/ws/handlers.py
```

---

## WEBSOCKET OVERVIEW

```yaml
websocket:
  endpoint: "ws://api/v1/ws"

  events:
    validation_started: "Validation started"
    layer_complete: "Layer finished"
    validation_complete: "All layers done"

  use_cases:
    - "Real-time progress during validation"
    - "Dashboard live updates"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# ws/manager.py
from typing import Dict, Set
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        if user_id not in self._connections:
            self._connections[user_id] = set()
        self._connections[user_id].add(websocket)

    async def broadcast_to_user(self, user_id: str, message: dict):
        if user_id in self._connections:
            for ws in self._connections[user_id]:
                await ws.send_json(message)

manager = ConnectionManager()
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: "ls src/hl_mcp/cloud/ws/manager.py"
```

---

## REFERÊNCIA

- `./S28_CONTEXT.md` - Cloud API Base
- `./S31_CONTEXT.md` - Dashboard Setup

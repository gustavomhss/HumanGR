# S38 - cockpit-visual | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S38
  name: cockpit-visual
  title: "Cockpit Visual"
  wave: W4-Growth
  priority: P2-MEDIUM
  type: implementation

objective: "Dashboard visual real-time para validações"

dependencies:
  - S33  # Dashboard Components

deliverables:
  - dashboard/app/cockpit/page.tsx
  - dashboard/app/cockpit/live-view.tsx
  - dashboard/components/cockpit/layer-progress.tsx
  - dashboard/components/cockpit/finding-stream.tsx
```

---

## COCKPIT OVERVIEW

```yaml
cockpit:
  description: "Real-time visualization of validation progress"

  features:
    - "Live layer progress (7 layers)"
    - "Finding stream as they appear"
    - "Veto level indicator"
    - "Triple redundancy status"
    - "Time remaining estimate"

  visuals:
    layer_progress: "7 horizontal bars filling as layers complete"
    finding_stream: "Live feed of findings"
    veto_indicator: "Color-coded veto status"
    consensus_view: "3-column view for triple redundancy"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```typescript
// cockpit/live-view.tsx
'use client'

import { useEffect, useState } from 'react'
import { LayerProgress } from '@/components/cockpit/layer-progress'
import { FindingStream } from '@/components/cockpit/finding-stream'
import { VetoIndicator } from '@/components/cockpit/veto-indicator'

export function LiveView({ validationId }: { validationId: string }) {
  const [progress, setProgress] = useState<LayerProgress[]>([])
  const [findings, setFindings] = useState<Finding[]>([])
  const [vetoLevel, setVetoLevel] = useState<string>('NONE')

  useEffect(() => {
    const ws = new WebSocket(`ws://api/v1/ws/${validationId}`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'layer_complete') {
        setProgress(prev => [...prev, data.layer])
      }
      if (data.type === 'finding') {
        setFindings(prev => [data.finding, ...prev])
      }
      if (data.type === 'veto_update') {
        setVetoLevel(data.level)
      }
    }

    return () => ws.close()
  }, [validationId])

  return (
    <div className="grid grid-cols-3 gap-4">
      <LayerProgress layers={progress} />
      <FindingStream findings={findings} />
      <VetoIndicator level={vetoLevel} />
    </div>
  )
}
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls dashboard/app/cockpit/page.tsx
      ls dashboard/components/cockpit/layer-progress.tsx
```

---

## REFERÊNCIA

- `./S33_CONTEXT.md` - Dashboard Components
- `./S39_CONTEXT.md` - Analytics

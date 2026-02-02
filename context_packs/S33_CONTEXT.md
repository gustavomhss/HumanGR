# S33 - dashboard-components | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S33
  name: dashboard-components
  title: "Dashboard Components"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Componentes UI reutilizáveis"

dependencies:
  - S31  # Dashboard Setup

deliverables:
  - dashboard/components/ui/
  - dashboard/components/validation-card.tsx
  - dashboard/components/layer-badge.tsx
  - dashboard/components/finding-list.tsx
```

---

## COMPONENTS OVERVIEW

```yaml
components:
  ui/:
    - button.tsx
    - card.tsx
    - badge.tsx
    - table.tsx

  business/:
    - validation-card.tsx
    - layer-badge.tsx
    - finding-list.tsx
    - veto-indicator.tsx
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```typescript
// components/layer-badge.tsx
type VetoLevel = 'NONE' | 'WEAK' | 'MEDIUM' | 'STRONG'

const colors: Record<VetoLevel, string> = {
  NONE: 'bg-gray-100 text-gray-800',
  WEAK: 'bg-yellow-100 text-yellow-800',
  MEDIUM: 'bg-orange-100 text-orange-800',
  STRONG: 'bg-red-100 text-red-800',
}

export function LayerBadge({
  layer,
  veto
}: {
  layer: string
  veto: VetoLevel
}) {
  return (
    <span className={`px-2 py-1 rounded ${colors[veto]}`}>
      {layer} ({veto})
    </span>
  )
}
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls dashboard/components/ui/
      ls dashboard/components/validation-card.tsx
```

---

## REFERÊNCIA

- `./S31_CONTEXT.md` - Dashboard Setup
- `./S34_CONTEXT.md` - Billing Integration

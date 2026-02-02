# S32 - dashboard-pages | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S32
  name: dashboard-pages
  title: "Dashboard Pages"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Páginas principais do dashboard"

dependencies:
  - S31  # Dashboard Setup

deliverables:
  - dashboard/app/dashboard/page.tsx
  - dashboard/app/dashboard/validations/page.tsx
  - dashboard/app/dashboard/reports/[id]/page.tsx
  - dashboard/app/dashboard/settings/page.tsx
```

---

## PAGES OVERVIEW

```yaml
pages:
  /dashboard: "Overview com stats"
  /dashboard/validations: "Lista de validações"
  /dashboard/reports/[id]: "Report detalhado"
  /dashboard/settings: "Configurações"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```typescript
// dashboard/app/dashboard/page.tsx
export default function DashboardPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <div className="grid grid-cols-4 gap-4 mt-6">
        <StatCard title="Total Validations" value="1,234" />
        <StatCard title="Approved" value="89%" />
        <StatCard title="Rejected" value="11%" />
        <StatCard title="This Month" value="156" />
      </div>
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
      ls dashboard/app/dashboard/page.tsx
      ls dashboard/app/dashboard/validations/page.tsx
```

---

## REFERÊNCIA

- `./S31_CONTEXT.md` - Dashboard Setup
- `./S33_CONTEXT.md` - Dashboard Components

# S31 - dashboard-setup | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S31
  name: dashboard-setup
  title: "Dashboard Setup"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Next.js dashboard setup"

dependencies:
  - S29  # Cloud API Endpoints

deliverables:
  - dashboard/package.json
  - dashboard/next.config.js
  - dashboard/app/layout.tsx
  - dashboard/app/page.tsx
```

---

## DASHBOARD STACK

```yaml
frontend:
  framework: "Next.js 14 (App Router)"
  styling: "Tailwind CSS"
  components: "shadcn/ui"
  state: "React Query"
  auth: "@clerk/nextjs"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```typescript
// dashboard/app/layout.tsx
import { ClerkProvider } from '@clerk/nextjs'
import './globals.css'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>{children}</body>
      </html>
    </ClerkProvider>
  )
}
```

```typescript
// dashboard/app/page.tsx
import { auth } from '@clerk/nextjs'
import { redirect } from 'next/navigation'

export default async function Home() {
  const { userId } = auth()

  if (!userId) {
    redirect('/sign-in')
  }

  redirect('/dashboard')
}
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls dashboard/package.json
      ls dashboard/app/layout.tsx

  G1_BUILD_WORKS:
    validation: "cd dashboard && npm run build"
```

---

## REFERÊNCIA

- `./S29_CONTEXT.md` - Cloud API
- `./S32_CONTEXT.md` - Dashboard Pages

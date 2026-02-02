# S40 - team-features | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W4-Growth"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S40
  name: team-features
  title: "Team Features"
  wave: W4-Growth
  priority: P2-MEDIUM
  type: implementation

objective: "Features de equipe (multi-user, roles)"

dependencies:
  - S27  # Authentication

deliverables:
  - src/hl_mcp/cloud/teams/__init__.py
  - src/hl_mcp/cloud/teams/models.py
  - src/hl_mcp/cloud/teams/service.py
  - src/hl_mcp/cloud/teams/roles.py
  - src/hl_mcp/cloud/api/routes/teams.py
  - dashboard/app/dashboard/team/page.tsx
```

---

## TEAM FEATURES

```yaml
teams:
  features:
    - "Create/manage organizations"
    - "Invite team members"
    - "Role-based access (admin, member, viewer)"
    - "Team-wide validation history"
    - "Shared configurations"

  roles:
    admin:
      - "Full access"
      - "Manage members"
      - "Manage billing"
      - "Configure layers"

    member:
      - "Run validations"
      - "View history"
      - "View reports"

    viewer:
      - "View history"
      - "View reports"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# teams/roles.py
from enum import Enum


class TeamRole(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


ROLE_PERMISSIONS = {
    TeamRole.ADMIN: [
        "validations:create",
        "validations:read",
        "reports:read",
        "team:manage",
        "billing:manage",
        "config:manage",
    ],
    TeamRole.MEMBER: [
        "validations:create",
        "validations:read",
        "reports:read",
    ],
    TeamRole.VIEWER: [
        "validations:read",
        "reports:read",
    ],
}


def has_permission(role: TeamRole, permission: str) -> bool:
    """Check if role has permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])
```

```python
# teams/service.py
class TeamService:
    """Team management service."""

    async def create_organization(
        self,
        name: str,
        owner_id: str,
    ) -> Organization:
        """Create new organization."""
        pass

    async def invite_member(
        self,
        org_id: str,
        email: str,
        role: TeamRole,
    ) -> Invitation:
        """Invite member to organization."""
        pass

    async def remove_member(
        self,
        org_id: str,
        user_id: str,
    ) -> None:
        """Remove member from organization."""
        pass

    async def change_role(
        self,
        org_id: str,
        user_id: str,
        new_role: TeamRole,
    ) -> None:
        """Change member's role."""
        pass
```

---

## 🎉 MILESTONE: GROWTH COMPLETE

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                    🚀 GROWTH FEATURES COMPLETE 🚀                  ║
║                                                                    ║
║    Wave 4: Growth ✅                                               ║
║                                                                    ║
║    Features:                                                       ║
║    ✅ 6 Perspectives (tired_user, malicious_insider, etc)          ║
║    ✅ CI/CD Integrations (GitHub Action, GitLab CI)                ║
║    ✅ Cockpit Visual (real-time validation view)                   ║
║    ✅ Analytics & Trends                                           ║
║    ✅ Team Features (multi-user, roles)                            ║
║                                                                    ║
║    ALL 41 SPRINTS COMPLETE!                                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/teams/models.py
      ls src/hl_mcp/cloud/teams/roles.py
      ls src/hl_mcp/cloud/api/routes/teams.py

  G1_ROLES_DEFINED:
    validation: |
      python -c "
      from hl_mcp.cloud.teams.roles import TeamRole, ROLE_PERMISSIONS
      assert len(TeamRole) == 3
      assert TeamRole.ADMIN in ROLE_PERMISSIONS
      "
```

---

## REFERÊNCIA

- `./S27_CONTEXT.md` - Authentication
- `../SPRINT_INDEX.yaml` - Complete sprint index

---

## ALL WAVES COMPLETE

```yaml
summary:
  Wave_0_Foundation:
    sprints: "S00-S02"
    status: "✅ Complete"

  Wave_1_CoreEngine:
    sprints: "S03-S14"
    status: "✅ Complete"

  Wave_2_OSSRelease:
    sprints: "S15-S24"
    status: "✅ Complete"
    milestone: "M1 - OSS Launch"

  Wave_3_CloudMVP:
    sprints: "S25-S35"
    status: "✅ Complete"
    milestone: "M2 - Cloud MVP"

  Wave_4_Growth:
    sprints: "S36-S40"
    status: "✅ Complete"
    milestone: "M3 - Growth"

total_sprints: 41
total_context_packs: 41
```

# Data Model: User Onboarding and Authentication

## Entities

### User (Better Auth Standard + Extensions)

The core user entity managed by Better Auth, extended with onboarding data.

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | String (UUID) | Yes | Primary Key |
| `name` | String | Yes | User's full name |
| `email` | String | Yes | Unique email address |
| `emailVerified` | Boolean | Yes | Email verification status |
| `image` | String | No | Profile picture URL |
| `createdAt` | DateTime | Yes | Creation timestamp |
| `updatedAt` | DateTime | Yes | Update timestamp |
| `softwareSkillLevel` | String | Yes | Enum: `Beginner`, `Intermediate`, `Expert` |
| `preferredOs` | String | Yes | Enum: `Windows`, `macOS`, `Linux`, `Other` |
| `hardwareEnvironment` | String | Yes | Enum: `Local`, `Cloud` |

### Session (Better Auth Standard)

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | String | Yes | Primary Key |
| `userId` | String | Yes | Foreign Key to User |
| `token` | String | Yes | Session token |
| `expiresAt` | DateTime | Yes | Expiration timestamp |
| `ipAddress` | String | No | Client IP |
| `userAgent` | String | No | Client User Agent |

### Account (Better Auth Standard - for OAuth)

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | String | Yes | Primary Key |
| `userId` | String | Yes | Foreign Key to User |
| `accountId` | String | Yes | Provider's account ID |
| `providerId` | String | Yes | e.g., "google", "github" |
| `accessToken` | String | No | OAuth access token |
| `refreshToken` | String | No | OAuth refresh token |
| `expiresAt` | DateTime | No | Token expiration |

### Verification (Better Auth Standard)

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | String | Yes | Primary Key |
| `identifier` | String | Yes | Email or Phone |
| `value` | String | Yes | OTP or link value |
| `expiresAt` | DateTime | Yes | Expiration timestamp |

## Schema Notes

-   **Database**: Neon (PostgreSQL).
-   **Management**: Better Auth CLI will handle migrations (`npx @better-auth/cli migrate`).
-   **Extensions**: The `softwareSkillLevel`, `preferredOs`, and `hardwareEnvironment` fields will be added to the `user` table via Better Auth's schema configuration.

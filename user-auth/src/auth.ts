import "dotenv/config";
import { betterAuth } from "better-auth";
import { Pool } from "pg";

export const auth = betterAuth({
    trustedOrigins: ["http://localhost:3001"],
    database: new Pool({
        connectionString: process.env.NEON_DATABASE_URL,
    }),
    emailAndPassword: {
        enabled: true,
    },
    user: {
        additionalFields: {
            softwareSkillLevel: {
                type: "string",
                required: false,
            },
            preferredOs: {
                type: "string",
                required: false,
            },
            hardwareEnvironment: {
                type: "string",
                required: false,
            },
        },
    },
});
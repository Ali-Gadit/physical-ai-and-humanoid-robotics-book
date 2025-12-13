import "dotenv/config";
import express from "express";
import cors from "cors";
import { toNodeHandler } from "better-auth/node";
import { auth } from "./auth.js";

const app = express();
const port = 3000;

const allowedOrigins = ["http://localhost:3001", "https://ali-gadit.github.io"];

const corsOptions = {
    origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            console.log("Blocked by CORS:", origin);
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
};

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

app.use("/api/auth", toNodeHandler(auth));

app.get("/health", (req, res) => {
  res.send("User Auth Service is running");
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});

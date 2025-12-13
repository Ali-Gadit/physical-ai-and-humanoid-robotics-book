import "dotenv/config";
import express from "express";
import cors from "cors";
import { toNodeHandler } from "better-auth/node";
import { auth } from "./auth.js";

const app = express();
const port = 3000;

app.use(cors({
    origin: ["http://localhost:3001", "https://ali-gadit.github.io"],
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
}));

app.use("/api/auth", toNodeHandler(auth));

app.get("/health", (req, res) => {
  res.send("User Auth Service is running");
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});

import "dotenv/config";
import express from "express";
import { toNodeHandler } from "better-auth/node";
import { auth } from "./auth";

const app = express();
const port = 3000;

app.all("/api/auth/*", toNodeHandler(auth));

app.get("/health", (req, res) => {
  res.send("User Auth Service is running");
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});

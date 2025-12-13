from fastapi import Request, HTTPException, Depends
from db.postgres import get_db_connection
import psycopg2
from typing import Optional

def get_current_user(request: Request):
    token = request.cookies.get("better-auth.session_token")
    if not token:
        # Also check Authorization header for Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Check session and get user
        cur.execute("""
            SELECT u.id, u.email, u.name, u."softwareSkillLevel", u."preferredOs", u."hardwareEnvironment"
            FROM session s
            JOIN "user" u ON s."userId" = u.id
            WHERE s.token = %s AND s."expiresAt" > NOW()
        """, (token,))
        user = cur.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
            
        return {
            "id": user[0],
            "email": user[1],
            "name": user[2],
            "softwareSkillLevel": user[3],
            "preferredOs": user[4],
            "hardwareEnvironment": user[5]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during auth check")
    finally:
        conn.close()

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.models.user import User
from app.security import get_current_user


def test_query_parameter_is_not_accepted_as_access_token():
    app = FastAPI()

    @app.get("/private")
    def private(user: User = Depends(get_current_user)):
        return {"user_id": user.id}

    response = TestClient(app).get("/private?access_token=invalid")

    assert response.status_code == 401

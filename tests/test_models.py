from sesame_ai.models import SignupResponse, LookupResponse, RefreshTokenResponse
from sesame_ai.exceptions import (
    SesameAIError, AuthenticationError, APIError, InvalidTokenError, NetworkError
)


def test_signup_response_parsing():
    data = {
        "kind": "identitytoolkit#SignupNewUserResponse",
        "idToken": "test_id_token",
        "refreshToken": "test_refresh_token",
        "expiresIn": "3600",
        "localId": "user123",
    }
    response = SignupResponse(data)
    assert response.id_token == "test_id_token"
    assert response.refresh_token == "test_refresh_token"
    assert response.local_id == "user123"
    assert response.expires_in == "3600"


def test_lookup_response_with_users():
    data = {
        "kind": "identitytoolkit#GetAccountInfoResponse",
        "users": [{"localId": "user123", "lastLoginAt": "1000", "createdAt": "900"}],
    }
    response = LookupResponse(data)
    assert response.kind == "identitytoolkit#GetAccountInfoResponse"
    assert response.local_id == "user123"


def test_lookup_response_no_users():
    data = {"kind": "identitytoolkit#GetAccountInfoResponse", "users": []}
    response = LookupResponse(data)
    assert response.kind == "identitytoolkit#GetAccountInfoResponse"


def test_refresh_token_response_parsing():
    data = {
        "access_token": "new_access",
        "id_token": "new_id",
        "refresh_token": "new_refresh",
        "expires_in": "3600",
        "token_type": "Bearer",
        "user_id": "user123",
        "project_id": "proj456",
    }
    response = RefreshTokenResponse(data)
    assert response.id_token == "new_id"
    assert response.refresh_token == "new_refresh"
    assert response.user_id == "user123"


def test_api_error():
    err = APIError(400, "Bad Request", [{"message": "detail"}])
    assert err.code == 400
    assert err.message == "Bad Request"
    assert "400" in str(err)
    assert len(err.errors) == 1


def test_api_error_no_details():
    err = APIError(500, "Internal Server Error")
    assert err.errors == []


def test_invalid_token_error():
    err = InvalidTokenError()
    assert isinstance(err, AuthenticationError)
    assert isinstance(err, SesameAIError)
    assert "Invalid" in str(err)


def test_exception_hierarchy():
    assert issubclass(AuthenticationError, SesameAIError)
    assert issubclass(APIError, SesameAIError)
    assert issubclass(InvalidTokenError, AuthenticationError)
    assert issubclass(NetworkError, SesameAIError)


def test_base_response_repr():
    data = {"idToken": "test", "refreshToken": "r", "expiresIn": "3600", "localId": "u"}
    response = SignupResponse(data)
    repr_str = repr(response)
    assert "SignupResponse" in repr_str

import hashlib

AUTHENTICATED = False
_PASSWORD_HASH = "26b1485b16923f1dba775b1e03bd10c85e5631b949a17fdb79227197e7c5b590"

def authenticate() -> bool:
    """执行身份验证，返回是否通过"""
    global AUTHENTICATED
    ans = input().strip()
    AUTHENTICATED = (hashlib.sha256(ans.encode()).hexdigest() == _PASSWORD_HASH)
    return AUTHENTICATED

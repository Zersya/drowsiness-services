from keycloak import KeycloakOpenID
import os
from dotenv import load_dotenv
import logging
import time

load_dotenv()

class KeycloakAuth:
    def __init__(self):
        try:
            self.keycloak_openid = KeycloakOpenID(
                server_url=os.getenv('KEYCLOAK_URL', 'https://sso.transtrack.id/'),
                client_id=os.getenv('KEYCLOAK_CLIENT_ID', 'drowsiness-detections-dashboard'),
                realm_name=os.getenv('KEYCLOAK_REALM', 'internal'),
                client_secret_key=os.getenv('KEYCLOAK_CLIENT_SECRET', 'eFXnpqyV2NTFXbiiMGvRgpJZqEp0gATL')
            )
        except Exception as e:
            logging.error(f"Failed to initialize Keycloak client: {str(e)}")
            raise

    def authenticate(self, username, password):
        try:
            token = self.keycloak_openid.token(
                username=username,
                password=password
            )
            user_info = self.keycloak_openid.userinfo(token['access_token'])
            return {
                'success': True,
                'token': token,
                'user_info': user_info
            }
        except Exception as e:
            logging.error(f"Authentication failed: {str(e)}")
            return {
                'success': False,
                'error': 'Invalid credentials'
            }

    def verify_token(self, token):
        try:
            token_info = self.keycloak_openid.introspect(token)
            return token_info.get('active', False)
        except Exception as e:
            logging.error(f"Token verification failed: {str(e)}")
            return False

    def refresh_token(self, refresh_token):
        try:
            new_token = self.keycloak_openid.refresh_token(refresh_token)
            return {
                'success': True,
                'token': new_token
            }
        except Exception as e:
            logging.error(f"Token refresh failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def logout(self, refresh_token):
        try:
            self.keycloak_openid.logout(refresh_token)
            return True
        except Exception as e:
            logging.error(f"Logout failed: {str(e)}")
            return False

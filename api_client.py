import requests
import logging
import os
import datetime
from urllib.parse import urljoin

class ApiClient:
    def __init__(self, base_url, api_endpoint, api_token, username, password):
        self.base_url = base_url
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
    
    def update_token(self, new_token):
        """Updates the API token."""
        self.api_token = new_token
        
    def perform_login(self):
        """Performs login and returns new token."""
        login_url = urljoin(self.base_url, "/vss/user/login.action")
        
        login_data = {
            'username': self.username,
            'password': self.password,
            'lang': 'en',
            'platform': 'web',
            'version': 'v2'
        }
        
        headers = {
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'platform': 'web',
            'version': 'v2'
        }
        
        try:
            response = requests.post(login_url, data=login_data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if response_data.get('status') == 10000:  # Success status
                new_token = response_data.get('data', {}).get('token')
                if new_token:
                    # Update the token
                    self.update_token(new_token)
                    # Optionally save to .env file for persistence
                    self.update_env_file('API_TOKEN', new_token)
                    self.logger.info("Successfully logged in and updated token")
                    return True
            
            self.logger.error(f"Login failed: {response_data.get('msg')}")
            return False
            
        except Exception as e:
            self.logger.error(f"Login error: {e}")
            return False
    
    def update_env_file(self, key, value):
        """Updates a single value in the .env file."""
        try:
            # Read existing .env file
            if os.path.exists('.env'):
                with open('.env', 'r') as file:
                    lines = file.readlines()
            else:
                lines = []

            # Find and replace the line with the key
            key_found = False
            new_lines = []
            for line in lines:
                if line.startswith(f'{key}='):
                    new_lines.append(f'{key}={value}\n')
                    key_found = True
                else:
                    new_lines.append(line)

            # Add the key if it wasn't found
            if not key_found:
                new_lines.append(f'{key}={value}\n')

            # Write back to .env file
            with open('.env', 'w') as file:
                file.writelines(new_lines)
                
        except Exception as e:
            self.logger.error(f"Error updating .env file: {e}")
    
    def fetch_video_evidence(self, start_time, end_time, retry_count=0):
        """Fetches video evidence from the API with session handling."""
        if retry_count >= 3:  # Limit retry attempts
            self.logger.error("Maximum retry attempts reached")
            return {'status': 'error', 'error': 'Maximum retry attempts reached'}
            
        self.logger.info(f"Fetching video evidence from {start_time} to {end_time}")
        formatted_start_time = start_time.strftime('%Y-%m-%d+%H:%M:%S')
        formatted_end_time = end_time.strftime('%Y-%m-%d+%H:%M:%S')

        data = {
            'alarmType': '',
            'startTime': formatted_start_time,
            'endTime': formatted_end_time,
            'pageNum': '1',
            'pageSize': 30,
            'token': self.api_token,
            'scheme': 'https',
            'lang': 'en'
        }
        
        headers = {
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'platform': 'web',
            'version': 'v2'
        }

        try:
            response = requests.post(self.api_endpoint, headers=headers, data=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Handle session expiration
            if response_data.get('status') == 10023:
                self.logger.info("Session expired. Attempting to login again...")
                if self.perform_login():
                    # Retry the fetch with new token
                    return self.fetch_video_evidence(start_time, end_time, retry_count + 1)
                else:
                    return {'status': 'error', 'error': 'Failed to refresh session'}
            
            # Check if we have valid data
            if response_data.get('status') == 10000:  # Success status
                evidence_list = response_data.get('data', {}).get('list', [])
                if evidence_list:
                    self.logger.info(f"Found {len(evidence_list)} evidence items")
                    return {'status': 'success', 'data': evidence_list}
                
            self.logger.info("No evidence found in the response")
            return {'status': 'no_data', 'data': []}
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching video evidence: {e}")
            return {'status': 'error', 'error': str(e)}
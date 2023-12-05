from __future__ import print_function

import os
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
import numpy

SCOPES = ["https://www.googleapis.com/auth/drive"]


#Google Drive 인증 절차
creds = None
#인증 토큰 재활용
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#인증 토큰 재발급
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
service = build('drive', 'v3', credentials=creds)


files = []
page_token = None

folder_response = service.files().list(q="mimeType = 'application/vnd.google-apps.folder' and name = 'Project'",spaces='drive', pageSize=100, fields="nextPageToken, files(id, name)").execute()
folder_result = folder_response.get('files',[])
folder_id = folder_result[0].get('id')

while True:
    file_response = service.files().list(q="mimeType = 'image/png' and '"+folder_id+"' in parents",spaces='drive', pageSize=10, fields="nextPageToken, files(id, name)",pageToken = page_token).execute()
    file_result = file_response.get('files',[])
    page_token = file_response.get('nextPageToken', None)
    
    for file in file_result:
        file_id = file.get('id')
        file_request = service.files().get_media(fileId=file_id)
        
        #FILE STREAM OPEN
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, file_request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        file_stream.seek(0)
        file_contents = file_stream.read()
        
        baseImage = cv2.imdecode(numpy.fromstring(file_contents, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.getcwd()+"\\original\\"+file.get('name'),baseImage)
        cropImage = baseImage[128:384,128:384]
        cv2.imwrite(os.getcwd()+"\\crop\\"+file.get('name'),cropImage)
        #cv2.imshow("test",cropImage)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        service.files().delete(fileId=file_id).execute()
    if page_token is None :
        break

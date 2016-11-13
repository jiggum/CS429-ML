# -*- coding: utf-8

import requests
import time
import json
from bs4 import BeautifulSoup
from datetime import  timedelta, date
import urllib
from StringIO import StringIO
from PIL import Image

def get_event_infos(html ,session):
    soup = BeautifulSoup("".join(html), "html.parser")
    table = soup.find('table',{"class":"req_tbl_02"})
    rows = table.findAll('tr')


    try:
        event_title = rows[0].findAll('td')[0].text.strip()
        event_day = rows[1].findAll('td')[0].text.strip()
        event_place = rows[1].findAll('td')[1].text.strip()
        event_writer = rows[2].findAll('td')[0].text.strip()
        event_createdPoint = rows[2].findAll('td')[1].text.strip()
        event_text = "<p>" + "</p><p>".join(p.text for p in rows[4].findAll('td')[0].findAll('p')) + "</p>"
        event_img = rows[4].findAll('td')[0].find('img')

    except:
        event_title = ""
        event_day = ""
        event_place = ""
        event_writer = ""
        event_createdPoint = ""
        event_text = ""
        event_img = ""
    try:
        event_img = event_img['src']
    except:
        event_img = ""

    ctx = {
        'event_title' : event_title,
        'event_day' : event_day,
        'event_place' : event_place,
        'event_writer' : event_writer,
        'event_createdPoint' : event_createdPoint,
        'event_text' : event_text,
        'event_img' : event_img,
    }
    return ctx


my_id = ''
my_pw = ''
with open('my_account.txt', 'r') as f:
    my_id = f.readline().strip()
    my_pw = f.readline().strip()
htmls = []

session = requests.Session()
cur_time = int(time.time())
return_url = 'portal.kaist.ac.kr/user/ssoLoginProcess.face?timestamp=' + str(cur_time - 30);
return_url = '?returnURL=' + urllib.quote(return_url, safe='~()*!.\'') + \
             ' &timestamp=' + str(cur_time)
base_url = 'https://portalsso.kaist.ac.kr/ssoProcess.ps'
r = session.post(base_url + return_url, data={'userId': my_id, 'password': my_pw})

data = {
    'boardId': 'seminar_events',
    'start': (date.today() + timedelta(days=0)).isoformat(),
    'end': (date.today() + timedelta(days=300)).isoformat(),
}


r = session.post('https://portal.kaist.ac.kr/board/scheduleList.brd', data=data)
event_list =json.loads(r.text)['Data'];
event_id_list = list(i['scheduleId'] for i in event_list)
for event_id in event_id_list:
    htmls.append(session.get('https://portal.kaist.ac.kr/board/read.brd?cmd=READ&boardId=seminar_events&bltnNo=' + event_id + '&lang_knd=ko&userAgent=Chrome&isMobile=false&'))

i = 257
for html in htmls:
    info = get_event_infos(html,session)
    url = info['event_img']
    if url.startswith('/board'):
        url = 'https://portal.kaist.ac.kr' + url
    print url
    if url:
        try:
            r = session.get(url)
            image = Image.open(StringIO(r.content)).convert('RGB')
            image.save("./posters/" + str(i)+".jpg")
            i+=1
        except:
            print "failed from", url


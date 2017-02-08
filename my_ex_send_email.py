# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:42:09 2017

@author: jiang_y
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
http://www.scottcking.com

Quickie walkthrough on sending an e-mail or SMS/text message
using GMail's SMTP server and Python.  This is 100% cross platform
and will work in Windows, Mac, Linux.

This code passes Pylint with a perfect 10.0 score.
http://www.pylint.org

Extensive list of SMS gateways:
http://www.quertime.com/article/arn-2010-11-04-1-complete-list-of-email-to-sms-gateways/

Enjoy!

'''


import smtplib


def send_email(title = 'hello', message = 'test message'):
    ''' Send an e-mail or SMS text via GMail SMTP '''

    gmail_username = 'tensorflow4ml@gmail.com'       # sender's gmail username
    gmail_password = 'TAI88ping'                 # sender's gmail password
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    msg_from = 'tensorflow4ml@gmail.com'         # who the message is 'from'
    msg_subject = title           # message subject
    msg_to = 'yong.j.jiang@gmail.com'    # ex: 18005558989@tmomail.net   phone_number@your_sms_gateway
    msg_text = message

    # If intent is SMS/text, remove/comment the header subject line
    # If intent is e-mail, add/uncomment the header subject line
    # A comment in python is a '#' symbol placed at the beginning of the line

    headers = ['From: {}'.format(msg_from),
               'Subject: {}'.format(msg_subject),
               'To: {}'.format(msg_to),
               'MIME-Version: 1.0',
               'Content-Type: text/html']

    msg_body = '\r\n'.join(headers) + '\r\n\r\n' + msg_text

    session = smtplib.SMTP(smtp_server, smtp_port)
    session.ehlo()
    session.starttls()
    session.ehlo()
    session.login(gmail_username, gmail_password)
    session.sendmail(msg_from, msg_to, msg_body)
    session.quit()

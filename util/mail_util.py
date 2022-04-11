#!/usr/bin/python
# coding:utf-8

import sys

import time
import smtplib
from email.mime.text import MIMEText


class email_sender_calss():
    def send_email(self):
        sender = 'udw9109727@163.com'
        receiver = '171913146@qq.com'
        smtpserver = 'smtp.163.com'
        username = 'udw9109727@163.com'
        password = 'MZOQLNCGFQXJXVED'
        subject = '我发送的邮件主题'
        str_html = 'finished_{}'.format(time.asctime(time.localtime()))
        # 信息
        msg = MIMEText(str_html, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['from'] = sender
        msg['to'] = receiver
        smtp = smtplib.SMTP(smtpserver)

        # smtp.esmtp_features["auth"] = "nonoy"
        (code, resp) = smtp.login(username, password)
        if 0:
            print("fail")
        else:
            print("success")
            result = smtp.sendmail(sender, receiver, msg.as_string())
            print(result)
            smtp.quit()
        pass


def finished_mail():
    app = email_sender_calss()
    app.send_email()


if __name__ == '__main__':
    finished_mail()
    print(time.asctime(time.localtime()))

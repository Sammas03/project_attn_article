#!/usr/bin/python
# coding:utf-8

import sys

import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header

class email_sender_calss():
    def send_email(self):
        sender = 'udw9109727@163.com'
        receiver = '171913146@qq.com'
        smtpserver = 'smtp.163.com'
        username = 'udw9109727@163.com'
        password = 'MZOQLNCGFQXJXVED'
        subject = '模型训练完成٩(๑`^´๑)۶'
        local_time = time.asctime(time.localtime())
        # 信息
        mail_msg = f"""
        <h2>深度学习模型已经训练完成啦！</h2>
        <p>{local_time}</p>
        <p><img src="cid:image1"></p>
        """

        msgRoot = MIMEMultipart('related')
        msgRoot['from'] = sender
        msgRoot['to'] = receiver
        msgRoot['Subject'] = subject
        msgAlternative = MIMEMultipart('alternative')
        msgRoot.attach(msgAlternative)
        msgAlternative.attach(MIMEText(mail_msg, 'html', 'utf-8'))


        #引入图片
        # 指定图片为当前目录
        import os
        current_path = os.path.dirname(__file__)
        fp = open(current_path+'/pic/mail.jpg', 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()

        # 定义图片 ID，在 HTML 文本中引用
        msgImage.add_header('Content-ID', '<image1>')
        msgRoot.attach(msgImage)

        # 连接smtp服务端
        smtp = smtplib.SMTP(smtpserver)

        # smtp.esmtp_features["auth"] = "nonoy"
        (code, resp) = smtp.login(username, password)
        if 0:
            print("fail")
        else:
            print("success")
            result = smtp.sendmail(sender, receiver, msgRoot.as_string())
            print(result)
            smtp.quit()
        pass


def finished_mail():
    app = email_sender_calss()
    app.send_email()


if __name__ == '__main__':
    finished_mail()
    print(time.asctime(time.localtime()))

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path

def send_start(sim_name = 'Some Simulation', fig1 ='', fig2 =''):

    email = 'srvrinformer@gmail.com'
    password = '&Ab012_8Zp2!'
    send_to_email = 'jjovanovic996@gmail.com'
    subject = sim_name
    message = sim_name + ' has started computing!'

    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    # Setup the attachment
    file_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + fig1
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)

    # Setup the attachment
    file_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + fig2
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()

    return 0


def send_email(comp_time, fig1='', fig2='', fig3='', addmsg='', sim_name='Some Simulation'):

    H = int(comp_time / 3600)
    M = int((comp_time - H * 3600) / 60)
    S = int(comp_time - 3600 * H - 60 * M)

    email = 'srvrinformer@gmail.com'
    password = '&Ab012_8Zp2!'
    send_to_email = 'jjovanovic996@gmail.com'
    subject = sim_name
    message = 'Simulation Completed: ' + addmsg + ' Time taken: ' + str(H) + ':' + str(M) + ':' + str(S)

    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    # Setup the attachment
    file_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + fig1
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)

    # Setup the attachment
    file_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + fig2
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)

    # Setup the attachment
    file_location = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + fig3
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)


    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()

    return 0

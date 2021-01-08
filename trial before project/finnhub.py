# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
# import os
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail

# message = Mail(
#     from_email='adexong@gmail.com',
#     to_emails='adexong@gmail.com',
#     subject='Sending with Twilio SendGrid is Fun',
#     html_content='<strong>and easy to do anywhere, even with Python</strong>')
# try:
#     sg = SendGridAPIClient('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')
#     response = sg.send(message)
#     print(response.status_code)
#     print(response.body)
#     print(response.headers)
# except Exception as e:
#     print(e.message)


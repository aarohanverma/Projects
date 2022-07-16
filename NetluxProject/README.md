# This Script was developed as a screening task for Internship at Netlux Systems Private Limited

## The problem statement was that the company would receive emails for orders, and a dedicated employee would have to continuously keep on checking for emails for the same, maintain a spreadsheet for the customer information and SMS/Whatsapp the Product Key to the customers with confirmed order.

### Variables:

#### Set the generated 'password' for your account in the script under necessary credentials: (Refer to links below to generate it)
https://www.geeksforgeeks.org/fetch-unseen-emails-from-gmail-inbox/   
https://www.geeksforgeeks.org/how-to-read-emails-from-gmail-using-gmail-api-in-python/

https://www.geeksforgeeks.org/python-fetch-your-gmail-emails-from-a-particular-user/

#### Change the email to set the sender:
msgs = get_emails(search('FROM', 'verma.aarohan@gmail.com', con),con)     

#### Set your corresponding json file for ServiceAccountCredentials.from_json_keyfile_name('...',scope): (Refer to link below to generate it)
https://www.geeksforgeeks.org/using-google-sheets-as-database-in-python/  

#### Set your sheet name:
sheet = client.open("Netlux").sheet1                                      

### This Script is currently being used in production at Netlux Systems Private Limited



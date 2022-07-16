# This Script was developed as a screening task for Internship at Netlux Systems Private Limited

## The problem statement was that the company would receive emails for orders, and a dedicated employee would have to continuously keep on checking for emails for the same, maintain a spreadsheet for for customer information and SMS/Whatsapp the Product Key to the customers with confirmed order.

### Variables:
https://www.geeksforgeeks.org/fetch-unseen-emails-from-gmail-inbox/       //Set the generated 'password' for your account in the script under necessary credentials
msgs = get_emails(search('FROM', 'verma.aarohan@gmail.com', con),con)     //Change the email to set the sender
https://www.geeksforgeeks.org/using-google-sheets-as-database-in-python/  //Set your corresponding json file for ServiceAccountCredentials.from_json_keyfile_name('...',scope)
sheet = client.open("Netlux").sheet1                                      //Set your sheet name

This Script is currently being used in production at Netlux Systems Private Limited



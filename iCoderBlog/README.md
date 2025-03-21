# Django Blog Platform

## Table of Contents
- [Getting Started](#getting-started)
- [Project Overview](#project-overview)
- [Admin Credentials](#admin-credentials)
- [Features](#features)
- [Screenshot](#screenshot)
- [Usage](#usage)
- [License](#license)

---

## Getting Started

To run the project locally, execute the following command in your terminal:

```bash
python manage.py runserver
```

Make sure you have installed all necessary dependencies and that Django is properly configured.

---

## Project Overview

This project is a basic yet fully functional blog/forum web application built using **Django**. It provides:

- **CRUD Operations:** Create, Read, Update, and Delete blog posts.
- **User Authentication:** Users can sign up, log in, and manage their profiles.
- **Post Moderation:** Blog posts require approval from the Django admin before they are published and visible on the public Blog section.

This template is designed to serve as a customizable starting point for creating community forums, personal blogs, or content management systems.

---

## Admin Credentials

Use the following default credentials to access the Django admin panel:

- **Username:** `asus`
- **Password:** `12345`

> **Important:** Change these credentials before deploying the application in a production environment to ensure security.

---

## Features

- **User Authentication:** Secure sign-up and login processes.
- **Full CRUD Functionality:** Manage blog posts effortlessly.
- **Post Moderation:** Ensure quality content with an admin approval system.
- **Customizable Template:** Easily extend or modify the application to suit your specific needs.

---

## Screenshot

Below is a sample screenshot of the blog application:

![Blog Screenshot](https://user-images.githubusercontent.com/97247457/179349696-98600975-38fa-4f76-b6b7-7bdd447a3f93.png)

---

## Usage

After starting the server:

1. **Visit the Application:** Open your browser and navigate to `http://127.0.0.1:8000/`.
2. **User Registration & Login:** Sign up or log in to create new blog posts.
3. **Post Creation:** Submit a blog post, which will require admin approval.
4. **Post Approval:** Log in as an admin to approve pending posts, after which they will be published in the Blog section.

## License

This project is open-source. See the LICENSE file for more details.

---
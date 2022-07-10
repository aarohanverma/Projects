import email
from django.db import models

# Create your models here.

class Contact(models.Model):
    sn=models.AutoField(primary_key=True)
    name=models.CharField(max_length=255)
    phone=models.CharField(max_length=13)
    email=models.CharField(max_length=100)
    content=models.TextField()
    date=models.DateTimeField(auto_now_add=True, blank=True)

    def __str__(self) -> str:
        return 'Message From: ' + self.name + ' (' + self.email + ')'
from django.db import models
from django.template.defaultfilters import slugify
from datetime import datetime

# Create your models here.

class Post(models.Model):
    sn=models.AutoField(primary_key=True)
    title=models.CharField(max_length=255)
    author=models.CharField(max_length=255, blank=True, default="")
    content=models.TextField()
    slug=models.SlugField(max_length=200)
    date=models.DateTimeField(auto_now=True, blank=True)
    approved=models.BooleanField(default=False)

    def __str__(self) -> str:
        return self.title + ' by ' + self.author

    def save(self, *args, **kwargs):
        self.slug = slugify(self.title+' '+self.author+datetime.now().strftime("%H-%M-%S") )
        super(Post, self).save(*args, **kwargs)

    
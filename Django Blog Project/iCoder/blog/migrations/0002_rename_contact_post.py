# Generated by Django 4.0.5 on 2022-06-30 17:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Contact',
            new_name='Post',
        ),
    ]

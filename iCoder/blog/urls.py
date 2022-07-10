from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.blogHome, name='blogHome'),
    path('<str:slug>/', views.blogPost, name='blogPost'),
    path('<str:slug>/edit', views.blogPostEdit, name='blogPostEdit'),
    path('<str:slug>/deletepost', views.blogPostDelete, name='blogPostDelete'),
    path('<str:slug>/new', views.blogPostNew, name='blogPostNew'),
    path('<str:slug>/blogposts', views.myBlogPosts, name='myBlogPost')
]

from django.shortcuts import render, redirect
from django.http import HttpResponse
from blog.models import Post
from django.contrib import messages

# Create your views here.
def blogHome(request):
    allPosts = Post.objects.all()[0:5]
    params = {'allPosts':allPosts}
    return render(request, 'blog/blogHome.html', params)

def blogPost(request, slug):
    post=Post.objects.filter(slug=slug).first()
    params={'post':post}
    return render(request, 'blog/blogPost.html',params)

def myBlogPosts(request, slug):

    if request.method == "POST" and request.user.is_authenticated:
        title=request.POST['title']
        content=request.POST['content']
        post=Post(author=request.user.username,title=title,content=content,approved=False)
        post.save()
        messages.success(request, 'Your post was updated successfully!')
    allPosts = Post.objects.filter(author=slug)
    params = {'allPosts':allPosts}
    return render(request, 'blog/myBlogPosts.html', params)

def blogPostEdit(request, slug):
    post=Post.objects.filter(slug=slug).first()
    if request.user.is_authenticated and request.user.username == post.author:
        params={'post':post}
        post.delete()
        return render(request, 'blog/blogPostEdit.html',params)
    return HttpResponse("Not Allowed")

def blogPostDelete(request, slug):
    post=Post.objects.filter(slug=slug).first()
    if request.user.is_authenticated and request.user.username == post.author:
        post.delete()
        allPosts = Post.objects.filter(author=request.user.username)
        params = {'allPosts':allPosts}
        messages.success(request, 'Your post was deleted successfully!')
        return render(request, 'blog/myBlogPosts.html', params)
    return HttpResponse("Not Allowed")

def blogPostNew(request, slug):
    if request.user.is_authenticated:
        return render(request, 'blog/blogPostEdit.html')
    return HttpResponse("Not Allowed")



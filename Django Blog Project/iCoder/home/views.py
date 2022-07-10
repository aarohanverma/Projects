from django.shortcuts import render, redirect
from django.http import HttpResponse
from home.models import Contact
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from blog.models import Post
from django.contrib.auth.models import User

# Create your views here.
def home(request):
    return render(request, 'home/home.html')

def contact(request):
    
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        phone=request.POST['phone']
        content=request.POST['content']
        if len(name)<2 and len(email)<3 and len(phone)<6 and len(content)<2:
            messages.error(request, 'Please enter the details correctly!')
        else:
            contact=Contact(name=name,email=email,phone=phone,content=content)
            contact.save()
            messages.success(request, 'Your message has been successfult sent!')
    return render(request, 'home/contact.html')

def about(request):
    return render(request, 'home/about.html')

def search(request):
    query=request.GET['query']
    allPosts=Post.objects.filter(title__icontains=query)
    params={'allPosts':allPosts}
    return render(request, 'home/search.html', params)

def signup(request):
    if request.method == 'POST':
        username=request.POST['username']
        firstname=request.POST['firstname']
        lastname=request.POST['lastname']        
        signupemail=request.POST['signupemail']
        signuppassword=request.POST['signuppassword']
        confirmsignuppassword=request.POST['confirmsignuppassword']
        # check for erroneous input
        my_user=User.objects.create_user(username,signupemail,signuppassword)
        my_user.first_name=firstname
        my_user.last_name=lastname
        my_user.save()
        messages.success(request, 'Your account has been created successfully!')
        return redirect('/')
    else:
        return HttpResponse("Not Allowed")

def signin(request):
    if request.method == "POST":
        loginusername=request.POST['loginusername']
        loginpassword=request.POST['loginpassword']
        user=authenticate(username=loginusername,password=loginpassword)
        if user is not None:
            login(request,user)
            messages.success(request, 'Successfully Logged In!')
            return redirect('/')
        else:
            messages.error(request, 'Invalid Credentials!')
            return redirect('/')

    return HttpResponse("Not Allowed")

def signout(request):
    if request.user.is_authenticated:
        logout(request)
        messages.success(request, 'Logged out Successfully!')
        return redirect('/')
    return HttpResponse("Not Allowed")
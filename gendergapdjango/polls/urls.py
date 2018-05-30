from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('247-1.html', views.explainer, name='explainer'),
    path('247-2.html', views.explainer2, name='explainer2'),
    path('submit_rule', views.submit_rule, name='submit_rule'),
    path('submit_list', views.submit_list, name='submit_list')
]

from django.http import HttpResponse, JsonResponse
from . import mlt
from django.shortcuts import render
from django.shortcuts import render_to_response

def index(request):
    return(HttpResponse("Hello - you're at the polls index"))

def explainer(request):
    return render_to_response('247-1.html')

def explainer2(request):
    return render_to_response('247-2.html')

def submit_rule(request):
    # if this is a post request we need to process the rule
    if request.method == 'POST':
        dict = request.POST
        rule = dict['rule']
        correct = mlt.model.submit_rule(rule)
        if correct:
            return redirect('http://web.stanford.edu/~alicezhy/SexismInTech/247-3-1.html')
        else:
            return redirect('http://web.stanford.edu/~alicezhy/SexismInTech/247-3-2.html')
    else:
        return render(request, 'rule_form.html')

def submit_list(request):
    if request.method == 'POST':
        dic = request.post
        lis = dic['list']
        correct = mlt.model.submit_list(lis)
        return JsonResponse({'correct_list': str(correct)})
    else:
        return HttpResponse('please use a post request')

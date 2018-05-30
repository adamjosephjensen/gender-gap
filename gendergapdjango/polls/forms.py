from django import forms

class RuleForm(forms.Form):
    rule_guess = forms.CharField(label='Your guess', max_length=200)

from django import forms

class KeyForm(forms.Form):
    key = forms.CharField()

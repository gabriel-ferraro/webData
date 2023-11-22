from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField


class MLForm(FlaskForm):
    classifier = SelectField('Classificador', choices=[('knn', 'KNN'), ('mlp', 'MLP'), ('dt', 'Decision Tree'), ('rf', 'Random Forest')])
    param1 = StringField('Parâmetro 1')
    param2 = StringField('Parâmetro 2')
    param3 = StringField('Parâmetro 3')
    submit = SubmitField('Treinar')


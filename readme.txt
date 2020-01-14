Запуск:

Установка необходимых пакетов:
pip install -r requirements.txt -v  

Создать файлы для тренировки:
python model -f Zapolnennoe_po_faktoram.xlsx [-v : для валидации]

Обучение сети:
python model -t -m [имя модели, default=model] -n [номер критерия от 1 до 6] [-c : продолжить обучение]

Использование обученной сети:
Для валидации:
python model -v
Предсказание для неоцененных образцов:
python model -p [папка, в которой лежат образцы]

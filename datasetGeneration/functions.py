import random
import string
from PIL import Image
from random_word import RandomWords
import numpy as np

def generate_email():
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase
    only_letters = string.ascii_lowercase + string.ascii_uppercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase
    username = ''.join(random.choice(letters) for i in range(random.randint(4, 10)))
    username2 = ''.join(random.choice(letters) for i in range(random.randint(4, 10)))
    domain_name = ''.join(random.choice(letters) for i in range(random.randint(5, 10)))
    domain_extension = ''.join(random.choice(only_letters) for i in range(random.randint(2, 3)))
    domain = domain_name + '.' + domain_extension
    email = username + random.choice(['.', '_', '']) + username + '@' + domain
    return email

def generate_word():
    r = RandomWords()
    return r.get_random_word()
 
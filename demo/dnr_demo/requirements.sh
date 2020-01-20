set -e
apt-get update
apt-get install build-essential wget git python3-pip apache2 apache2-utils ssl-cert libapache2-mod-wsgi-py3 supervisor redis-server 
a2enconf mod-wsgi
service apache2 restart

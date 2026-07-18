class url_model:
    def __init__(self, 
                 scheme, 
                 user,
                 password,
                 host, 
                 port=None, 
                 path='/', 
                 query='', 
                 fragment='', 
                 raw='',
                 normalized=''
                 ):
        self.raw = raw
        self.scheme = (scheme or '').lower()
        self.user = user
        self.password = password
        self.host = (host or '').lower()
        self.port = port
        self.path = path
        self.query = query
        self.fragment = fragment
        self.normalized= normalized


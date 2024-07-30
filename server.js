const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3000;
const LOCAL_IP = '192.168.4.55'; // Replace with your local IP address
const ESP32_IP = 'http://192.168.4.31'; // Replace with your ESP32 IP address

const options = {
    key: fs.readFileSync('localhost-key.pem'),
    cert: fs.readFileSync('localhost.pem')
};

app.use(express.static(path.join(__dirname, 'public')));

// Proxy API requests to the ESP32
app.use('/servo', createProxyMiddleware({
    target: ESP32_IP,
    changeOrigin: true,
    pathRewrite: {
        '^/servo': '/servo'
    },
    onProxyReq: (proxyReq, req, res) => {
        console.log('Proxying request to:', proxyReq.path);
    },
    onProxyRes: (proxyRes, req, res) => {
        let body = '';
        proxyRes.on('data', (chunk) => {
            body += chunk;
        });
        proxyRes.on('end', () => {
            console.log('Response from ESP32:', body);
        });
    },
    onError: (err, req, res) => {
        console.error('Proxy error:', err);
        res.status(500).send('Proxy error');
    }
}));

// Catch-all to prevent unnecessary root requests
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

https.createServer(options, app).listen(PORT, LOCAL_IP, () => {
    console.log(`Server is running on https://${LOCAL_IP}:${PORT}`);
});

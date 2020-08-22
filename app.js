const morgan = require('morgan')
const express = require('express')
const { PythonShell } = require("python-shell");

const app = express()

PORT = process.env.PORT || 3000;

// Handling CORS
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, Authorization"
    );
    if (req.method === "OPTIONS") {
        res.header("Access-Control-Allow-Methods", "PUT, POST, PATCH, DELETE, GET");
        return res.status(200).json({
            status: true,
            message: "request granted",
        });
    }
    next();
});

// Middlewares
app.use(morgan("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    res.sendFile('index.html', { root: 'detection-website' })
})

app.post('/test', (req, res) => {
    res.send('Hello World')
})

var python_process;
app.post('/start', function (req, res) {

    var pyshell = new PythonShell('detect.py');

    pyshell.end(function (err) {
        if (err) {
            console.log(err);
        }
    });
    python_process = pyshell.childProcess;

    res.send('Started.');
});

app.post('/stop', function (req, res) {
    python_process.kill('SIGINT');
    res.send('Stopped');
});

// Throw 404 error if the request does not match an existing route
app.use((req, res, next) => {
    const error = new Error();
    error.status = 404;
    error.message = "404 route not found";
    next(error);
});

//  Return the error thrown by any part of the project.
app.use((err, req, res, next) => {
    res.status(err.status || 404).json({
        status: false,
        error: err.message,
    });
});

app.listen(PORT, () => console.log(`\nServer running on localhost:${PORT}\n`));
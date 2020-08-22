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

app.get('/test', (req, res) => {
    res.send('Hello World')
})

var python_process;
app.post('/start', function (req, res) {
    const input = req.body.input || 'videos/cctv_footage.mp4';
    const output = req.body.output || 'output/Test.avi';
    const yolo = req.body.yolo || 'yolo-coco';
    const confidence = req.body.confidence || 0.0;
    const threshold = req.body.threshold || 0.0;


    let options = {
        mode: "text",
        args: ["--input", input, "--output", output, '--yolo', yolo, '--confidence', confidence, 'threshold', threshold]
    };
    var pyshell = new PythonShell('detect.py', options);

    pyshell.end(function (err) {
        if (err)
            console.log(err);
    });
    python_process = pyshell.childProcess;

    res.sendFile('index.html', { root: 'detection-website' });
});

app.post('/stop', function (req, res) {
    python_process.kill('SIGINT');
    res.sendFile('index.html', { root: 'detection-website' });
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
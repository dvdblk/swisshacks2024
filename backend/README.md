# Backend code

## Quickstart
```
docker compose up --build
```

## "Design"
API was supposed to invoke inference for 4 models. Two parallel pipelines:

<img src="../docs/audio-sample-flow.png" width="40%" height="40%">

* only the left branch is implemented in the API

## Preview

<video src="../docs/backend-preview.mp4" width="400" />

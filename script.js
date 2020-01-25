const imageUpload = document.getElementById('imageUpload')

Promise.all([
    faceapi.nets.faceRecongnitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('models')
]).then(start)

async function start() {
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)
    const labledFaceDescriptor = await loadLabeledImages()
    const faceMatcher = new faceapi.faceMatcher(labledFaceDescriptor, 0.6)
    let image, canvas

    document.body.append('โหลดเสร็จแล้ว')

    imageUpload.addEventListener('change', async() => {
        image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)
        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)
        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const result = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        result.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvas)
        })
    })
}

function loadLabeledImages() {
    const labels = ['Jisoo', 'Jennie', 'Lisa', 'Rose']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= labels.length; i++) {
                const img = await faceapi.fetchImage('')
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}
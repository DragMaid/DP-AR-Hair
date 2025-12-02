import React, { useState } from "react";
import api from "@/api/api";

export default function ImageUploadBox() {
    const [currentImage, setCurrentImage] = useState<any | null>(null);

    const uploadHandler = (event: React.ChangeEvent<HTMLInputElement>) => {
        const image = event.target.files?.[0];
        if (!image) return;

        const data = new FormData();
        data.append("file", image);

        api.post("/image", data)
            .then(res => {
                setCurrentImage(res.data);
            });
    };

    return (
        <div>
            <div>
                <input type="file" name="file" onChange={uploadHandler} />
            </div>
            <img src={`http://localhost:8080/image/${currentImage}`} />
        </div>
    );
}

import React from "react";

export default function ImageUploadBox() {
    const uploadHandler = (event: React.ChangeEvent<HTMLInputElement>) => {
        console.log("User uploaded: ", event.target.files);
    };

    return (
        <div>
            <input type="file" name="file" onChange={uploadHandler}></input>
        </div>
    )
}
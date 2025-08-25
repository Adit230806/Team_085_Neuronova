# Secure ML Pipeline Frontend

A modern React frontend for the Secure ML Pipeline with advanced data visualization, real-time charts, and intuitive user interface.

## Features

- **Modern UI**: Built with React, TailwindCSS, and responsive design
- **Interactive Charts**: Chart.js integration for data visualization
- **Real-time Updates**: Live pipeline status and results
- **Security Dashboard**: Blockchain-style audit logs visualization
- **File Upload**: Drag-and-drop interface for CSV/JSON files
- **JWT Authentication**: Secure token-based authentication
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm start
   ```

3. **Access the Application**
   - Frontend: http://localhost:3000
   - Make sure backend is running at http://127.0.0.1:8000

## Default Login Credentials

- **Username**: admin
- **Password**: admin123

## Features Overview

### 🏠 Home Page
- System status monitoring
- Feature overview cards
- Quick action buttons
- Security features showcase

### 📁 File Upload
- Drag-and-drop file upload
- Support for CSV and JSON files
- File validation and size limits
- Real-time upload progress

### ⚙️ Pipeline Controls  
- Algorithm selection (Isolation Forest / LOF)
- Interactive configuration
- Real-time pipeline status
- Execution history

### 📊 Results & Visualization
- Interactive data tables with pagination
- Multiple chart types (Pie, Bar, Scatter)
- Anomaly highlighting
- Export capabilities

### 🔒 Security Logs
- Blockchain-style audit trail
- Cryptographic hash verification
- Real-time security events
- Tamper-proof logging

## Component Structure

```
src/
├── components/           # React components
│   ├── HomePage.js      # Landing page
│   ├── LoginPage.js     # Authentication
│   ├── FileUploadPage.js # File upload interface
│   ├── PipelineControlsPage.js # ML pipeline controls
│   ├── ResultsPage.js   # Data visualization
│   ├── SecurityLogsPage.js # Audit logs
│   └── Navbar.js        # Navigation
├── context/             # React contexts
│   ├── AuthContext.js   # Authentication state
│   └── NotificationContext.js # Toast notifications
├── services/            # API services
│   └── api.js          # Axios configuration
├── App.js              # Main application
├── index.js            # Entry point
└── index.css           # Global styles
```

## API Integration

The frontend communicates with the FastAPI backend through RESTful APIs:

- **Authentication**: `/auth/login`
- **File Upload**: `/upload`
- **Pipeline Execution**: `/pipeline/run`
- **Results**: `/results`
- **Security Logs**: `/security/logs`

## Styling & Design

- **TailwindCSS**: Utility-first CSS framework
- **Custom Components**: Reusable button and card styles
- **Responsive Design**: Mobile-first approach
- **Color Scheme**: Professional blue/gray palette
- **Animations**: Subtle hover effects and loading states

## Chart Types

### Pie Chart
- Anomaly distribution visualization
- Normal vs Anomaly ratio

### Bar Chart
- Comparative anomaly counts
- Easy-to-read vertical bars

### Scatter Plot
- Two-dimensional data plotting
- Anomaly points highlighted in red
- Normal points shown in green

## Security Features

### JWT Token Management
- Automatic token storage
- Token expiration handling
- Secure API communication

### Input Validation
- File type restrictions
- Size limit enforcement
- Data format validation

### Error Handling
- User-friendly error messages
- Network error recovery
- Authentication error handling

## Build & Deployment

### Development Build
```bash
npm start
```

### Production Build
```bash
npm run build
```

### Testing
```bash
npm test
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Performance Optimizations

- Component lazy loading
- Chart.js lazy initialization
- Optimized bundle size
- Efficient re-rendering

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend is running with CORS enabled
   - Check proxy configuration in package.json

2. **Authentication Issues**
   - Clear localStorage
   - Check token expiration
   - Verify backend connectivity

3. **Chart Not Loading**
   - Check Chart.js imports
   - Verify data format
   - Check console for errors

### Debug Mode

Set `REACT_APP_DEBUG=true` for additional logging:

```bash
REACT_APP_DEBUG=true npm start
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

This project is part of the Secure ML Pipeline system for educational and demonstration purposes.